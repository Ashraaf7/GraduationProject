using System;
using System.Collections.Generic;
using System.Data;
using System.Data.Entity;
using System.Linq;
using System.Net;
using System.Web;
using System.Web.Mvc;
using CourseApp1.data;
using Microsoft.AspNet.Identity;
using PagedList;

namespace CourseApp1.Controllers
{
    public class CoursController : Controller
    {
        private courses_dbEntities db = new courses_dbEntities();

        // GET: Cours
        public ActionResult Index(string search, int? Category_id, int? Trainer_id, int? page)
        {
            ViewBag.Category_id = new SelectList(db.Categories, "ID", "Name");
            ViewBag.Trainer_id = new SelectList(db.Trainers, "ID", "Name");
            var courses = db.Courses.Include(c => c.Category).Include(c => c.Trainer);
            
            if (!String.IsNullOrEmpty(search))
            {
                courses = courses.Where(s => s.Name.Contains(search) || s.Category.Name.Contains(search));

            }
            if (Category_id != null)
            {
                courses = courses.Where(c => c.Category_id == Category_id);
            }
            if (Trainer_id != null)
            {
                courses = courses.Where(t => t.Trainer_id == Trainer_id);
            }
            return View(courses.ToList().ToPagedList(page ?? 1, 8));
        }
        public ActionResult CatCourses(int? id, int? page)
        {
            if (id == null)
            {
                return new HttpStatusCodeResult(HttpStatusCode.BadRequest);
            }
            var courses = db.Courses.Include(c => c.Category).Include(c => c.Trainer).Where(c=>c.Category_id==id);
            return View(courses.ToList().ToPagedList(page ?? 1, 8));
        }
        
        // GET: Cours/Details/5
        public ActionResult Details(int? id)
        {
            if (id == null)
            {
                return new HttpStatusCodeResult(HttpStatusCode.BadRequest);
            }
            Cours cours = db.Courses.Find(id);
            if (cours == null)
            {
                return HttpNotFound();
            }
            if (User.Identity.IsAuthenticated)
            {
                if (db.Favourites.ToList().Any(c => c.CourseId == id && c.UserId == User.Identity.GetUserId()))
                {
                    ViewBag.Added = "added";
                }
            }
            
            return View(cours);
        }
        [HttpPost]
        [ValidateAntiForgeryToken]
        [Authorize]
        public ActionResult Details(Favourite favourite, int? id)
        {
            if (id != null)
            {
                favourite.CourseId = (int)id;
            }
            favourite.UserId = User.Identity.GetUserId();
            if (ModelState.IsValid)
            {
                if(db.Favourites.ToList().Any(c=>c.CourseId == id && c.UserId == favourite.UserId))
                {
                    TempData["message"] = "Already Added to Favourites";
                }
                else
                {
                    db.Favourites.Add(favourite);
                    db.SaveChanges();
                }                
                
                return RedirectToAction("Details", "Cours", new { id = id });
            }
            return View();
        }
        [Authorize]
        public ActionResult Favourites()
        {
            var user= User.Identity.GetUserId();
            var favourites = db.Favourites.Include(f => f.AspNetUser).Include(f => f.Cours).Where(c=>c.UserId == user).ToList();
            return View(favourites);
        }
        //to remove course from favourites
        [HttpPost]
        public JsonResult DeleteUserJ(int Id)
        {
            try
            {
                
                    var User = db.Favourites.Find(Id);         //fetching the user with Id   
                    if (User != null)
                    {
                        db.Favourites.Remove(User);              //deleting from db  
                        db.SaveChanges();
                        return Json(User, JsonRequestBehavior.AllowGet);
                    }
                    return Json(null, JsonRequestBehavior.AllowGet);
                
            }

            catch (Exception ex)
            {
                return null;
            }
        }
        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                db.Dispose();
            }
            base.Dispose(disposing);
        }
    }
}
