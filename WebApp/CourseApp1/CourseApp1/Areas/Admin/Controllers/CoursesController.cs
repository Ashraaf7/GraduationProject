using System;
using System.Collections.Generic;
using System.Data;
using System.Data.Entity;
using System.IO;
using System.Linq;
using System.Net;
using System.Web;
using System.Web.Mvc;
using CourseApp1.data;
using PagedList;

namespace CourseApp1.Areas.Admin.Controllers
{
    [Authorize(Roles ="Admin")]
    public class CoursesController : Controller
    {
        private courses_dbEntities db = new courses_dbEntities();

        // GET: Admin/Courses
        public ActionResult Index(string search,int? Category_id,int? Trainer_id,int? page)
        {
            ViewBag.Category_id = new SelectList(db.Categories, "ID", "Name");
            ViewBag.Trainer_id = new SelectList(db.Trainers, "ID", "Name");
            var courses = db.Courses.Include(c => c.Category).Include(c => c.Trainer);            
            if (!String.IsNullOrEmpty(search))
            {
                courses=courses.Where(s => s.Name.Contains(search)|| s.Category.Name.Contains(search));
                
            }
            if (Category_id != null)
            {
                courses = courses.Where(c => c.Category_id == Category_id);
            }
            if(Trainer_id != null)
            {
                courses=courses.Where(t=>t.Trainer_id == Trainer_id);
            }
            return View(courses.ToList().ToPagedList(page ?? 1, 3));
        }
        public ActionResult TrainerCourses(int? id)
        {
            if (id == null)
            {
                return new HttpStatusCodeResult(HttpStatusCode.BadRequest);
            }
            var courses = db.Courses.Include(c => c.Category).Include(c => c.Trainer).Where(t=>t.Trainer_id==id);
            return View(courses.ToList());
        }
        public ActionResult test(string search, int? Category_id, int? Trainer_id)
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
            return View(courses.ToList());
        }

        // GET: Admin/Courses/Details/5
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
            if (cours.IsYoutube == true)
            {
                ViewBag.message = "true";
            }
            return View(cours);
        }

        // GET: Admin/Courses/Create
        public ActionResult Create()
        {
            
            ViewBag.Category_id = new SelectList(db.Categories, "ID", "Name");
            ViewBag.Trainer_id = new SelectList(db.Trainers, "ID", "Name");
            return View();
        }

        // POST: Admin/Courses/Create
        // To protect from overposting attacks, enable the specific properties you want to bind to, for 
        // more details see https://go.microsoft.com/fwlink/?LinkId=317598.
        [HttpPost]
        [ValidateAntiForgeryToken]
        public ActionResult Create([Bind(Include = "ID,Name,Creation_Date,Description,Category_id,Trainer_id,CourseLink,IsYoutube,ImagePath,ImageFile")] Cours cours)
        {

            if (ModelState.IsValid)
            {
                cours.Creation_Date= DateTime.Now;
                if (cours.ImageFile != null)
                {
                    var fileextension=Path.GetExtension(cours.ImageFile.FileName);
                    var imgguid=Guid.NewGuid().ToString();
                    cours.ImgPath=imgguid+fileextension;
                    //save file
                    string filepath=Server.MapPath($"~/Uploads/Courses/{cours.ImgPath}");
                    cours.ImageFile.SaveAs(filepath);
                }
                db.Courses.Add(cours);
                db.SaveChanges();               
                return RedirectToAction("Index");
            }

            ViewBag.Category_id = new SelectList(db.Categories, "ID", "Name", cours.Category_id);
            ViewBag.Trainer_id = new SelectList(db.Trainers, "ID", "Name", cours.Trainer_id);
            return View(cours);
        }

        // GET: Admin/Courses/Edit/5
        public ActionResult Edit(int? id)
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
            ViewBag.Category_id = new SelectList(db.Categories, "ID", "Name", cours.Category_id);
            ViewBag.Trainer_id = new SelectList(db.Trainers, "ID", "Name", cours.Trainer_id);
            return View(cours);
        }

        // POST: Admin/Courses/Edit/5
        // To protect from overposting attacks, enable the specific properties you want to bind to, for 
        // more details see https://go.microsoft.com/fwlink/?LinkId=317598.
        [HttpPost]
        [ValidateAntiForgeryToken]
        public ActionResult Edit([Bind(Include = "ID,Name,Creation_Date,Description,Category_id,Trainer_id,CourseLink,,ImagePath,ImageFile")] Cours cours)
        {
            if (ModelState.IsValid)
            {
                if (cours.ImageFile != null)
                {
                    var fileextension = Path.GetExtension(cours.ImageFile.FileName);
                    var imgguid = Guid.NewGuid().ToString();
                    cours.ImgPath = imgguid + fileextension;
                    //save file
                    string filepath = Server.MapPath($"~/Uploads/Courses/{cours.ImgPath}");
                    cours.ImageFile.SaveAs(filepath);
                }
                db.Entry(cours).State = EntityState.Modified;
                if (cours.ImageFile == null)
                {
                    db.Entry(cours).Property(x => x.ImgPath).IsModified = false;
                }
                db.SaveChanges();
                return RedirectToAction("Index");
            }
            ViewBag.Category_id = new SelectList(db.Categories, "ID", "Name", cours.Category_id);
            ViewBag.Trainer_id = new SelectList(db.Trainers, "ID", "Name", cours.Trainer_id);
            return View(cours);
        }

        // GET: Admin/Courses/Delete/5
        public ActionResult Delete(int? id)
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
            return View(cours);
        }

        // POST: Admin/Courses/Delete/5
        [HttpPost, ActionName("Delete")]
        [ValidateAntiForgeryToken]
        public ActionResult DeleteConfirmed(int id)
        {
            Cours cours = db.Courses.Find(id);
            db.Courses.Remove(cours);
            db.SaveChanges();
            return RedirectToAction("Index");
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
