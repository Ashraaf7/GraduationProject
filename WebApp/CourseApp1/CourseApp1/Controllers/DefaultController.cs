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

namespace CourseApp1.Controllers
{
    public class DefaultController : Controller
    {
        private courses_dbEntities db = new courses_dbEntities();
        // GET: Default
        public ActionResult Index()
        {
            ViewBag.Trainer_id = new SelectList(db.Trainers, "ID", "Name");
            var courses = db.Courses.Include(c => c.Category).Include(c => c.Trainer);
            var categories = db.Categories.Include(c => c.Category1).ToList();
            ViewBag.Categories = categories;
            return View(courses.ToList().Take(10));
        }
        public ActionResult Home()
        {
            var courses = db.Courses.Include(c => c.Category).Include(c => c.Trainer);
            return View(courses.ToList());
        }
        public ActionResult home2()
        {
            return View();
        }
    }
}